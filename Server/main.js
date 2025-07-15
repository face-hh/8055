const express = require('express');
const multer = require('multer');
const WebSocket = require('ws');
const http = require('http');
const axios = require('axios');
const FormData = require('form-data');

const app = express();
const server = http.createServer(app);

const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 20 * 1024 * 1024
    }
});

const wss = new WebSocket.Server({ server: server });
const questClients = new Set();

const DIFFUSION_SERVERS = [
    'http://localhost:8002',
    'http://localhost:8003',
    'http://localhost:8004',
    'http://localhost:8005',
    'http://localhost:8006',
    'http://localhost:8007',
    'http://localhost:8008',
    'http://localhost:8009',
];

const GAN_SERVER = 'http://localhost:8010';

let currentServerIndex = 0;
let serverHealthy = new Array(DIFFUSION_SERVERS.length).fill(true);

function getNextDiffusionServer() {
    let attempts = 0;
    while (attempts < DIFFUSION_SERVERS.length) {
        const server = DIFFUSION_SERVERS[currentServerIndex];
        const isHealthy = serverHealthy[currentServerIndex];

        currentServerIndex = (currentServerIndex + 1) % DIFFUSION_SERVERS.length;

        if (isHealthy) {
            return { url: server, index: currentServerIndex - 1 };
        }
        attempts++;
    }
    return { url: DIFFUSION_SERVERS[0], index: 0 };
}

async function healthCheck() {
    for (let i = 0; i < DIFFUSION_SERVERS.length; i++) {
        try {
            await axios.get(`${DIFFUSION_SERVERS[i]}/health`, { timeout: 1000 });
            serverHealthy[i] = true;
        } catch (e) {
            serverHealthy[i] = false;
        }
    }
}

setInterval(healthCheck, 5000);
healthCheck();

wss.on('connection', (ws) => {
    ws.on('message', (message) => {
        try {
            const data = JSON.parse(message.toString());
            if (data.type === 'quest_client') {
                questClients.add(ws);
            }
        } catch (e) {
            // Ignore non-JSON messages
        }
    });

    ws.on('close', () => {
        questClients.delete(ws);
    });

    ws.on('error', (error) => {
        questClients.delete(ws);
    });
});

app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    next();
});

app.post('/upload', upload.single('image'), async (req, res) => {
    if (!req.file) {
        res.status(200).end();
        return;
    }
    if (questClients.size === 0) {
        res.status(200).end();
        return;
    }

    let mode = 'diffusion';
    if (req.query.mode === 'gan' || req.headers['x-anime-mode'] === 'gan') {
        mode = 'gan';
    }

    try {
        let serverUrl;
        let serverIndex = -1;
        if (mode === 'gan') {
            serverUrl = GAN_SERVER;
        } else {
            const result = getNextDiffusionServer();
            serverUrl = result.url;
            serverIndex = result.index;
        }

        const formData = new FormData();
        formData.append('image', req.file.buffer, {
            filename: 'frame.jpg',
            contentType: 'image/jpeg'
        });

        const startTime = Date.now();

        const response = await axios.post(`${serverUrl}/process`, formData, {
            headers: {
                ...formData.getHeaders()
            },
            responseType: 'arraybuffer',
            timeout: 15000
        });

        const processTime = Date.now() - startTime;
        const processedBuffer = Buffer.from(response.data);

        let sentCount = 0;
        questClients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                try {
                    client.send(processedBuffer, { binary: true });
                    sentCount++;
                } catch (e) {
                    questClients.delete(client);
                }
            } else {
                questClients.delete(client);
            }
        });

    } catch (e) {
        if (e.code === 'ECONNREFUSED' || e.code === 'ETIMEDOUT') {
            if (mode === 'diffusion') {
                const failedIndex = DIFFUSION_SERVERS.findIndex(server => e.config?.url?.includes(server));
                if (failedIndex !== -1) {
                    serverHealthy[failedIndex] = false;
                }
            }
        }
    }
    res.status(200).end();
});

const PORT = process.env.PORT || 3035;
server.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
    console.log(`Quest URL: http://135.125.163.131:${PORT}/upload`);
    console.log(`Diffusion servers: ${DIFFUSION_SERVERS.join(', ')}`);
    console.log(`GAN server: ${GAN_SERVER}`);
});

process.on('SIGINT', () => {
    server.close(() => process.exit(0));
});