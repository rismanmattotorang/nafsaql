import json, time, urllib.request, sys
URL="http://127.0.0.1:18081/v1/chat/completions"
def call(body):
    req=urllib.request.Request(URL,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.time(); r=json.load(urllib.request.urlopen(req,timeout=600)); return r, time.time()-t0
q="A train leaves at 3:40 pm and arrives at 6:05 pm the same day. How many minutes is the trip? Reply with the number only."
# 1) fast path: thinking off, n=4 samples, logprobs
r,dt=call({"model":"qwen3-1.7b","messages":[{"role":"user","content":q}],"max_tokens":16,"temperature":0.7,"n":4,"logprobs":True,"top_logprobs":3,"chat_template_kwargs":{"enable_thinking":False}})
ans=[c["message"]["content"].strip() for c in r["choices"]]
lp=[sum(t["logprob"] for t in c["logprobs"]["content"])/max(1,len(c["logprobs"]["content"])) for c in r["choices"] if c.get("logprobs")]
print(json.dumps({"probe":"fast_n4_logprobs","wall_s":round(dt,2),"answers":ans,"mean_logprob":[round(x,3) for x in lp],"usage":r.get("usage")}))
# 2) slow path: thinking on, budget 256 (server-side --reasoning-budget), reasoning separated
r,dt=call({"model":"qwen3-1.7b","messages":[{"role":"user","content":q}],"max_tokens":400,"temperature":0.0,"chat_template_kwargs":{"enable_thinking":True}})
m=r["choices"][0]["message"]
print(json.dumps({"probe":"slow_budget256","wall_s":round(dt,2),"reasoning_chars":len(m.get("reasoning_content") or ""),"answer":(m.get("content") or "").strip()[:120],"usage":r.get("usage"),"finish":r["choices"][0].get("finish_reason")}))
