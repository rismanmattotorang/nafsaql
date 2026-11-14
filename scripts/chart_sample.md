## Decode speed by model

```chart
{"type":"bar","title":"Decode speed","subtitle":"tokens per second, median of three runs","xLabel":"Model","yLabel":"tok/s","data":[{"model":"Quick 1.7B","decode":15.5,"prompt":94.1},{"model":"Careful 4B","decode":6.7,"prompt":38.2},{"model":"Embedder 0.6B","decode":0,"prompt":180.6}],"keys":["decode","prompt"],"indexBy":"model"}
```

```chart
{"type":"line","title":"Switch cost","subtitle":"seconds, before and after the shadow","xLabel":"Turn","yLabel":"s","series":[{"id":"Cold","data":[{"x":1,"y":37},{"x":2,"y":36},{"x":3,"y":38},{"x":4,"y":37}]},{"id":"Shadow","data":[{"x":1,"y":2.4},{"x":2,"y":2.0},{"x":3,"y":2.1},{"x":4,"y":1.9}]}]}
```

```chart
{"type":"donut","title":"Where a reply's time goes","unit":"%","data":[{"label":"Prompt read","value":38},{"label":"Decode","value":47},{"label":"Check","value":9},{"label":"Other","value":6}]}
```

```chart
{"type":"radar","title":"Quick vs careful","indexBy":"metric","keys":["Quick","Careful"],"data":[{"metric":"Speed","Quick":9,"Careful":4},{"metric":"Maths","Quick":5,"Careful":9},{"metric":"Code","Quick":6,"Careful":8},{"metric":"Recall","Quick":7,"Careful":7},{"metric":"Memory","Quick":8,"Careful":5}]}
```

```chart
{"type":"heatmap","title":"Right answers by domain","data":[{"id":"Quick","data":[{"x":"Maths","y":62},{"x":"Code","y":71},{"x":"Facts","y":80}]},{"id":"Careful","data":[{"x":"Maths","y":84},{"x":"Code","y":88},{"x":"Facts","y":85}]}]}
```

```chart
{"type":"treemap","title":"Disk use","unit":"GB","data":{"name":"disk","children":[{"name":"Models","children":[{"name":"Qwen3-4B","value":2.5},{"name":"Qwen3-1.7B","value":1.1},{"name":"Embedder","value":0.6}]},{"name":"Knowledge","children":[{"name":"Papers","value":0.9},{"name":"Vectors","value":0.4}]},{"name":"Logs","value":0.3}]}}
```

```chart
{"type":"sankey","title":"Where questions end up","nodes":[{"id":"Questions"},{"id":"Quick"},{"id":"Careful"},{"id":"Checked"},{"id":"Unchecked"}],"links":[{"source":"Questions","target":"Quick","value":70},{"source":"Questions","target":"Careful","value":30},{"source":"Quick","target":"Checked","value":40},{"source":"Quick","target":"Unchecked","value":30},{"source":"Careful","target":"Checked","value":25},{"source":"Careful","target":"Unchecked","value":5}]}
```

```chart
{"type":"scatter","title":"Length vs time","xLabel":"Tokens","yLabel":"Seconds","series":[{"id":"Replies","data":[{"x":40,"y":3},{"x":120,"y":9},{"x":200,"y":14},{"x":260,"y":19},{"x":90,"y":7},{"x":330,"y":25}]}]}
```

```chart
{"type":"funnel","title":"From question to verified","data":[{"id":"asked","label":"Asked","value":1000},{"id":"answered","label":"Answered","value":960},{"id":"checkable","label":"Checkable","value":640},{"id":"verified","label":"Verified","value":590}]}
```
