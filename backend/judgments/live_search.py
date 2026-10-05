"""Live public FJUD search. No search cache, credentials or CAPTCHA bypass."""
import re
import ssl
import time
from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin, urlparse, parse_qs
import requests
import truststore
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from backend.judgments.core import normalize, retrieve
from backend.judgments.issues import issue_profile, ISSUES

"""
用途：
- 將留言、摘要、BERT 分類轉成搜尋條件。
- 連線司法院裁判書系統。
- 解析搜尋結果和判決全文。
- 驗證網址只能來自司法院。
- 將取得的候選裁判重新排序。
其中 decisions() 雖然仍保留在檔案內，但現在的整合流程沒有使用它；BERT 標籤由新建的轉接服務直接轉換。
"""

class SystemTLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return super().init_poolmanager(*args, **kwargs)


def decisions(rows, bert=False):
    groups = {'騷擾': {'一般騷擾', '性騷擾', '侮辱貶抑', '威脅恐嚇'}, '霸凌': {'霸凌風險'}}
    result = []
    for name, labels in groups.items():
        matched = [r for r in rows if r['label'] in labels and (r.get('selected', False) if bert else True)]
        result.append({'label': name, 'positive': bool(matched),
                       'status': ('疑似' if bert else '詞彙初篩：疑似') + name if matched else '資訊不足／未檢出訊號',
                       'evidence': [r['label'] for r in matched]})
    return result


def search_plan(comment, context, results):
    text = comment + ' ' + context
    profile = issue_profile(text)
    predicted = {name for r in results if r['positive'] for name in r.get('evidence', [])}
    existing = {issue['label'] for issue in profile}
    profile.extend({**issue, 'matched': []} for issue in ISSUES
                   if issue['label'] in predicted and issue['label'] not in existing)
    features = list(dict.fromkeys(w for issue in profile for w in issue['matched']))[:6]
    behavior = list(dict.fromkeys(w for issue in profile for w in issue['behavior']))[:6]
    legal = list(dict.fromkeys(w for issue in profile for w in issue['legal']))[:6]
    labels = [r['label'] for r in results if r['positive']]
    # Predictions expand the search vocabulary, never gate all retrieval on literal labels.
    if '霸凌' in labels:
        legal += ['霸凌', '公然侮辱']
    if '騷擾' in labels and not legal:
        legal += ['騷擾', '跟蹤騷擾', '公然侮辱']
    legal = list(dict.fromkeys(legal))[:7]
    if not features:
        features = re.findall(r'[\u4e00-\u9fff]{2,8}', comment)[:2]
    network = '(留言+網路+私訊+訊息+社群+臉書+通訊軟體)'
    group = lambda words: '(' + '+'.join(words) + ')'
    queries = []
    if legal and features:
        queries.append({'stage': '留言用詞＋相關爭點', 'query': group(legal) + '&' + group(features)})
    if legal:
        queries.append({'stage': '相關爭點＋網路情境', 'query': group(legal) + '&' + network})
    if legal and behavior:
        queries.append({'stage': '相關爭點＋行為同義詞', 'query': group(legal) + '&' + group(behavior)})
    if not queries and features:
        queries.append({'stage': '未分類：留言用詞探索', 'query': group(features) + '&' + network})
    if any(len(q['query']) > 128 for q in queries):
        raise ValueError('搜尋條件超過官網限制，請縮短背景資訊。')
    return {'query': queries[0]['query'] if queries else '', 'queries': queries, 'features': features,
            'legal_terms': legal, 'behavior_terms': behavior, 'labels': labels, 'uncertain': not labels,
            'issues': [p['name'] for p in profile],
            'ranking_query': comment + ' ' + context + ' ' + ' '.join(legal + features)}


def rank_cases(plan, docs, k):
    candidates = retrieve(plan['ranking_query'], docs, max(len(docs), k))
    for hit in candidates:
        # Inspect the retrieved passage, not a distant incidental mention elsewhere.
        evidence = [w for w in plan.get('features', []) if w in hit['text']]
        behavior = [w for w in plan.get('behavior_terms', []) if w in hit['text']]
        issues = [w for w in plan.get('legal_terms', []) if w in hit['text'] or w in hit['doc']['title']]
        hit['match_terms'] = list(dict.fromkeys(evidence + behavior))
        hit['issue_terms'] = issues
        hit['match_level'] = '留言用詞命中' if evidence else '相同行為／爭點參考'
        offsets = [hit['text'].find(w) for w in (evidence or behavior)]
        preview_start = max(0, min(offsets) - 100) if offsets else 0
        hit['preview'] = ('…' if preview_start else '') + hit['text'][preview_start:preview_start + 420]
        if preview_start + 420 < len(hit['text']):
            hit['preview'] += '…'
        hit['score'] += min(len(evidence), 3) * .08 + min(len(behavior), 2) * .025
    if plan.get('legal_terms'):
        candidates = [h for h in candidates if h['issue_terms'] and h['match_terms']]
    return sorted(candidates, key=lambda h: -h['score'])[:k]


def safe_url(value, base='https://judgment.judicial.gov.tw/FJUD/default.aspx'):
    url = urljoin(base, value)
    parts = urlparse(url)
    if parts.scheme != 'https' or parts.hostname != 'judgment.judicial.gov.tw' or not parts.path.startswith('/FJUD/'):
        raise ValueError('裁判來源網址不在允許範圍。')
    return url


def parse_document(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    body = soup.select_one('.jud_content')
    jid = parse_qs(urlparse(url).query).get('id', [''])[0]
    if body is None or not jid:
        raise ValueError('未取得裁判全文；可能為驗證頁、移除資料或網站格式變更。')
    metadata = {}
    for row in soup.select('#jud > .row'):
        key, value = row.select_one('.col-th'), row.select_one('.col-td')
        if key and value:
            metadata[key.get_text(strip=True).rstrip('：')] = value.get_text(' ', strip=True)
    content = body.select_one('.htmlcontent') or body
    for tag in content.select('script,style'):
        tag.decompose()
    for br in content.select('br'):
        br.replace_with('\n')
    for block in content.select('div,p,tr'):
        block.append('\n')
    text = content.get_text('', strip=False).replace('\r', '').replace('\xa0', ' ')
    text = re.sub(r'\n[ \t]*\n+', '\n', text).strip()
    if len(text) < 80:
        raise ValueError('裁判正文過短，未列為完整判決。')
    case_name = metadata.get('裁判字號', '')
    court = re.split(r'\s*\d+\s*年', case_name)[0].strip()
    date_parts = jid.split(',')
    return normalize({'id': jid, 'court': court or None, 'date': date_parts[-2] if len(date_parts) >= 6 else '',
                      'title': metadata.get('裁判案由', case_name), 'text': text, 'source_url': url})


class LiveJudicialSearch:
    HOME = 'https://judgment.judicial.gov.tw/FJUD/default.aspx'

    def __init__(self):
        self.session = requests.Session()
        self.session.mount('https://judgment.judicial.gov.tw/', SystemTLSAdapter())
        self.session.headers['User-Agent'] = 'JudgmentResearchPrototype/0.2'

    def page(self, url, data=None):
        for attempt in range(2):
            try:
                response = self.session.request('POST' if data is not None else 'GET', safe_url(url),
                                                data=data, timeout=(8, 15), allow_redirects=False)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt == 0:
                    time.sleep(.5)
                    continue
                raise ValueError('司法院連線逾時或中斷，重試後仍未成功；請稍後再試。') from e
            if attempt == 0 and response.status_code in (500, 502, 503, 504):
                time.sleep(.5)
                continue
            break
        if response.status_code in (301, 302, 303, 307, 308):
            raise ValueError('司法院要求重新導向／驗證，請至官網確認後再查詢。')
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.text

    def find_links(self, query):
        soup = BeautifulSoup(self.page(self.HOME), 'html.parser')
        inputs = soup.select('input[name]')
        if not any(x['name'] == 'txtKW' for x in inputs):
            raise ValueError('查詢表單不可用；可能為維護或驗證頁。')
        payload = {x['name']: x.get('value', '') for x in inputs}
        payload['txtKW'] = query
        response = self.page(self.HOME, payload)
        soup = BeautifulSoup(response, 'html.parser')
        frame = soup.select_one('iframe[src*="qryresultlst.aspx"]')
        if frame is None:
            # Official no-results response embeds ErrorPage Q003 rather than a list.
            error_frame = soup.select_one('iframe#iframe-data[src]')
            if error_frame:
                target = urlparse(urljoin(self.HOME, error_frame['src']))
                if (target.scheme == 'https' and target.hostname == 'judgment.judicial.gov.tw'
                        and target.path == '/ErrorPage.aspx'
                        and parse_qs(target.query).get('err') == ['Q003']):
                    return []
            if any(term in response for term in ['查無資料', '查無符合', '查無裁判']):
                return []
            raise ValueError('未取得裁判列表；可能是網站維護、驗證或查詢格式改變。')
        response = self.page(safe_url(frame['src']))
        listing = BeautifulSoup(response, 'html.parser')
        links = list(dict.fromkeys(safe_url(a['href']) for a in listing.select('a[href*="data.aspx?"]')
                                  if parse_qs(urlparse(a['href']).query).get('ty') == ['JD']))
        if not links and not any(x in response for x in ['查無', '共 0', '共0']):
            raise ValueError('列表格式無法辨識，未以空結果取代連線異常。')
        return links

    def search(self, plan, k, progress=None, on_results=None):
        if not 1 <= k <= 10:
            raise ValueError('最多顯示判決數須為 1–10。')
        docs, warnings, attempts, seen = [], [], [], set()
        stages = plan.get('queries', [{'stage': '裁判查詢', 'query': plan['query']}])
        deadline = time.monotonic() + 90
        blocked = False
        try:
            for stage in stages:
                if blocked or len(seen) >= 20 or time.monotonic() > deadline:
                    break
                if progress:
                    progress('正在搜尋：' + stage['stage'])
                attempt = {**stage, 'candidates': 0, 'downloaded': 0, 'status': '搜尋中'}
                attempts.append(attempt)
                try:
                    links = self.find_links(stage['query'])
                    attempt['candidates'] = len(links)
                    attempt['status'] = '取得列表' if links else '沒有搜尋結果'
                    quota = 0
                    for url in links:
                        jid = parse_qs(urlparse(url).query).get('id', [''])[0]
                        if jid in seen:
                            continue
                        if quota >= max(5, k) or len(seen) >= 20 or time.monotonic() > deadline:
                            break
                        seen.add(jid)
                        quota += 1
                        if progress:
                            progress(f"{stage['stage']}：取得第 {quota} 筆全文")
                        time.sleep(.3)
                        try:
                            docs.append(parse_document(self.page(url), url))
                            attempt['downloaded'] += 1
                        except requests.HTTPError as e:
                            if e.response.status_code in (403, 429):
                                raise
                            warnings.append('部分裁判 HTTP 下載失敗，已略過。')
                        except (requests.RequestException, ValueError) as e:
                            warnings.append('部分全文無法取得：' + str(e)[:160])
                        if on_results:
                            on_results(rank_cases(plan, docs, k), len(docs))
                except requests.HTTPError as e:
                    attempt['status'] = f'HTTP {e.response.status_code}'
                    if e.response.status_code in (403, 429):
                        blocked = True
                        warnings.append('官網限制存取，已停止；不繞過驗證或流量限制。')
                    else:
                        warnings.append('官網查詢發生 HTTP 錯誤。')
                except (requests.RequestException, ValueError) as e:
                    attempt['status'] = '查詢失敗：' + str(e)[:160]
                    warnings.append(attempt['status'])
                hits = rank_cases(plan, docs, k)
                # Expand at least once so classification vocabulary cannot exclude all cases.
                if len(attempts) >= 2 and len(hits) >= k:
                    break
            if time.monotonic() > deadline:
                warnings.append('達到本次查詢時間預算，顯示已成功取得的資料。')
            failed = bool(attempts) and all(a['status'].startswith(('查詢失敗', 'HTTP')) for a in attempts)
            return {'hits': rank_cases(plan, docs, k), 'warnings': list(dict.fromkeys(warnings)),
                    'downloaded': len(docs), 'fetched_at': self.now(), 'attempts': attempts,
                    'error': '所有查詢均因連線或來源錯誤失敗；尚未完成裁判搜尋。' if failed else None}
        finally:
            self.session.close()

    @staticmethod
    def now():
        return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
