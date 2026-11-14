# nivo.js

`rust/crates/nafs-web/assets/nivo.js` is Nivo 0.99 (https://nivo.rocks) with React 18, bundled once so the chat page
needs no network and no build step at run time. `entry.jsx` is the only code of ours in it.

Rebuild (needs Node 20+ and the npm registry):

```bash
tools/nivo-bundle/build.sh
```

The chat page loads it on first use, when an answer holds a `chart` block, so a chat without charts never fetches it.
