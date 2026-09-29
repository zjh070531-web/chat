"""Single-file, incrementally saved HTML review of confirmed turn events."""
import base64
from html import escape
from pathlib import Path

import cv2


def video_time(seconds):
    millis = round(float(seconds) * 1000)
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    seconds, millis = divmod(millis, 1000)
    return f'{hours:02}:{minutes:02}:{seconds:02}.{millis:03}'


class EventReport:
    """Embed source-resolution full-frame screenshots; flush after each event."""
    def __init__(self, path, video_name):
        self.path = Path(path)
        self.file = self.path.open('x', encoding='utf-8')
        self.seen = set()
        self.closed = False
        self.file.write('''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>转头结果报告</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#182430;font:16px/1.65 system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:24px}h1{margin:0;font-size:28px}
.note{color:#526171}.toolbar{position:sticky;top:0;z-index:1;background:#fff;padding:12px 20px;border-bottom:1px solid #ddd}
input,button,a.download{font:inherit;padding:6px 12px;border:1px solid #b8c5d1;border-radius:6px;background:white;color:#183b57}
input{width:160px}article{background:#fff;border-radius:10px;padding:20px;margin:20px 0;box-shadow:0 2px 10px #183b5710}
article h2{margin:0;font-size:22px}article p{margin:6px 0 12px}article img{display:block;width:100%;height:auto;cursor:zoom-in;border:1px solid #dae1e8}
.download{display:inline-block;margin-top:10px;text-decoration:none}#zoom{max-width:96vw;max-height:96vh;overflow:auto;padding:16px;border:0;border-radius:8px}
#zoom img{max-width:none;width:auto;height:auto}#zoom button{position:sticky;top:0;left:0;display:block;margin-bottom:8px}
[hidden]{display:none!important}footer{margin:24px 0;color:#526171}
</style></head><body><main><h1>转头结果报告</h1>
''')
        self.file.write(f'<p class="note">视频：{escape(str(video_name))}</p>\n')
        self.file.write('''<p class="note">仅记录首次确认 TURNED 的事件，同一次持续转头不重复截图。
时间为视频内时间；截图保留原始完整画面和分辨率，红框标出该条记录对应的人。点击图片可查看原尺寸。</p>
<div class="toolbar"><label>人员 ID <input id="person-filter" placeholder="全部 / P16 / 16" oninput="filterEvents()"></label>
<span id="counter"></span> · <span id="run-status">处理中 / 尚未结束；刷新可查看已保存结果</span></div>
<p id="empty">尚无已确认的转头事件。</p>
<dialog id="zoom"><button onclick="this.parentElement.close()">关闭大图</button><img alt="完整分辨率截图"></dialog>
<script>
function filterEvents(){
 const q=document.getElementById('person-filter').value.trim().replace(/^p/i,'');
 const cards=[...document.querySelectorAll('article')]; let visible=0;
 cards.forEach(card=>{card.hidden=!!q && card.dataset.person!==q;if(!card.hidden)visible++;});
 document.getElementById('counter').textContent=`显示 ${visible} / 共 ${cards.length} 次`;
 document.getElementById('empty').hidden=visible>0;
 document.getElementById('empty').textContent=cards.length?'没有符合此 ID 的事件。':'尚无已确认的转头事件。';
}
document.addEventListener('click',event=>{
 if(event.target.matches('article img')){
  const dialog=document.getElementById('zoom');dialog.querySelector('img').src=event.target.src;dialog.showModal();
 }
 if(event.target.matches('a.download'))event.target.href=event.target.closest('article').querySelector('img').src;
});
document.addEventListener('DOMContentLoaded',filterEvents);
</script>
''')
        self.file.flush()

    def add(self, event, frame, person, frame_index):
        event_id = str(event['event_id'])
        if event_id in self.seen:
            return
        ident = int(event['person_id'])
        image = frame.copy()
        height, width = image.shape[:2]
        scale = max(1., width/1920)
        color = (40, 40, 255)
        for key in ('body_box', 'head_box'):
            box = person.get(key)
            if box is not None:
                x1, y1, x2, y2 = map(round, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, round(3*scale))
        box = person.get('head_box') or person.get('body_box')
        label = f'P{ident} TURNED  {video_time(event["confirmed_time"])}'
        x = max(4, round(box[0])) if box else 12
        y = max(round(32*scale), round(box[1])-round(12*scale)) if box else round(32*scale)
        font = .7*scale
        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, round(2*scale))
        x = max(4, min(x, width-text_width-4))
        y = min(height-4, y)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), round(5*scale), cv2.LINE_AA)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font, color, round(2*scale), cv2.LINE_AA)
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError('Cannot encode event screenshot')
        data = base64.b64encode(encoded).decode('ascii')
        number = len(self.seen)+1
        self.file.write(f'<article data-person="{ident}"><h2>{number}. P{ident} · {escape(event_id)}</h2>\n')
        self.file.write(f'<p>确认时间：<strong>{video_time(event["confirmed_time"])}</strong> '
                        f'（{float(event["confirmed_time"]):.3f} 秒） · '
                        f'转头起点：{video_time(event["start_time"])} · 帧号：{int(frame_index)} · '
                        f'完整图片：{width} × {height}</p>\n')
        self.file.write(f'<img loading="lazy" alt="P{ident} 转头确认时的完整画面" src="data:image/jpeg;base64,{data}">\n')
        self.file.write(f'<a class="download" download="P{ident}_frame_{int(frame_index)}.jpg" href="#">保存完整截图</a></article>\n')
        self.file.flush()
        self.seen.add(event_id)

    def finish(self, completed):
        if self.closed:
            return
        status = '处理完成' if completed else '处理已中断；以下为中断前保存的结果'
        self.file.write(f'<footer>{status}，共保存 {len(self.seen)} 次转头事件。</footer>\n'
                        f'<script>document.getElementById("run-status").textContent="{status}";filterEvents();</script>'
                        '</main></body></html>\n')
        self.file.close()
        self.closed = True
