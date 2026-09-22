/**
 * Crypto Signal Bot - Web Dashboard Server (Node.js)
 * Serves static dashboard + WebSocket for live updates.
 *
 * - Polls Python backend's data/recommendations.json every 10s
 * - Broadcasts to all connected clients via WebSocket
 * - Also exposes REST endpoints for recommendations & stats
 */
const express = require('express');
const WebSocket = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
require('dotenv').config({ path: path.resolve(__dirname, '../../../.env') });

const PORT = process.env.DASHBOARD_PORT || 8080;
const HOST = process.env.DASHBOARD_HOST || '0.0.0.0';
const RECOMMENDATIONS_FILE = path.resolve(__dirname, '../../../data/recommendations.json');
const POSITIONS_FILE = path.resolve(__dirname, '../../../data/open_positions.json');
const STATS_FILE = path.resolve(__dirname, '../../../data/daily_stats.json');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

// ---------- REST API ----------

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.get('/api/recommendations', (req, res) => {
  try {
    if (!fs.existsSync(RECOMMENDATIONS_FILE)) {
      return res.json({ timestamp: null, top_recommendations: [] });
    }
    const data = JSON.parse(fs.readFileSync(RECOMMENDATIONS_FILE, 'utf-8'));
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/positions', (req, res) => {
  try {
    if (!fs.existsSync(POSITIONS_FILE)) return res.json([]);
    res.json(JSON.parse(fs.readFileSync(POSITIONS_FILE, 'utf-8')));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get('/api/stats', (req, res) => {
  try {
    if (!fs.existsSync(STATS_FILE)) return res.json({});
    res.json(JSON.parse(fs.readFileSync(STATS_FILE, 'utf-8')));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ---------- HTTP + WS Server ----------

const server = http.createServer(app);
const wss = new WebSocket.Server({ server, path: '/ws' });

const clients = new Set();
wss.on('connection', (ws) => {
  clients.add(ws);
  console.log(`[WS] Client connected. Total: ${clients.size}`);
  ws.on('message', (msg) => {
    try {
      const data = JSON.parse(msg.toString());
      if (data.type === 'ping') ws.send(JSON.stringify({ type: 'pong' }));
    } catch (e) {}
  });
  ws.on('close', () => {
    clients.delete(ws);
    console.log(`[WS] Client disconnected. Total: ${clients.size}`);
  });
});

function broadcast(data) {
  const payload = JSON.stringify(data);
  for (const client of clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  }
}

// ---------- Poll recommendations file ----------

let lastMtime = 0;
function checkUpdates() {
  try {
    if (!fs.existsSync(RECOMMENDATIONS_FILE)) return;
    const stats = fs.statSync(RECOMMENDATIONS_FILE);
    if (stats.mtimeMs !== lastMtime) {
      lastMtime = stats.mtimeMs;
      const data = JSON.parse(fs.readFileSync(RECOMMENDATIONS_FILE, 'utf-8'));
      broadcast({
        type: 'recommendations',
        timestamp: new Date().toISOString(),
        data
      });
    }
  } catch (e) {
    console.error('[Poll error]', e.message);
  }
}

setInterval(checkUpdates, 5000);

// ---------- Start server ----------

server.listen(PORT, HOST, () => {
  console.log(`\n========================================`);
  console.log(`  Crypto Signal Bot - Dashboard`);
  console.log(`  Listening: http://${HOST}:${PORT}`);
  console.log(`  WebSocket: ws://${HOST}:${PORT}/ws`);
  console.log(`  Recommendations file: ${RECOMMENDATIONS_FILE}`);
  console.log(`========================================\n`);
});
