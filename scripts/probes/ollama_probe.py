import json, time, urllib.request, sys
def chat(model, prompt, think, max_tokens=256):
    body = {"model": model, "messages":[{"role":"user","content":prompt}], "stream": False,
            "think": think, "options": {"num_predict": max_tokens, "temperature": 0.0}}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t0=time.time(); r = json.load(urllib.request.urlopen(req, timeout=1800)); wall=time.time()-t0
    out = {"model": model, "think": think, "wall_s": round(wall,2),
           "prompt_tokens": r.get("prompt_eval_count"), "prompt_s": round(r.get("prompt_eval_duration",0)/1e9,2),
           "gen_tokens": r.get("eval_count"), "gen_s": round(r.get("eval_duration",0)/1e9,2),
           "load_s": round(r.get("load_duration",0)/1e9,2)}
    if out["prompt_s"]: out["prompt_tok_s"] = round(out["prompt_tokens"]/out["prompt_s"],1)
    if out["gen_s"]: out["gen_tok_s"] = round(out["gen_tokens"]/out["gen_s"],1)
    out["thinking_chars"] = len(r["message"].get("thinking","") or "")
    out["answer_head"] = (r["message"]["content"] or "")[:160].replace("\n"," ")
    return out
model = sys.argv[1]
q = "A train leaves at 3:40 pm and arrives at 6:05 pm the same day. How many minutes is the trip? Answer with the number only."
for think in (False, True):
    res = chat(model, q, think, 512 if think else 64)
    print(json.dumps(res)); sys.stdout.flush()
