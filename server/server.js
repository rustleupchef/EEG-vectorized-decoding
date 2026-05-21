const Mindwave = require('mindwave');
const http = require('http');
const fs = require('fs');
const path = require('path');

let latestMindwaveData = {
    eeg: null,
    time: null
};

const server = http.createServer((req, res) => {
    if (req.url === '/mindwave/data' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(latestMindwaveData));
    } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'error', message: 'Not Found' }));
    }
});

const PORT = 3000;
server.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
});

const mw = new Mindwave();
mw.on('eeg', data => {
    latestMindwaveData.eeg = data;
    latestMindwaveData.time = Date.now();
});

mw.connect('/dev/rfcomm0');