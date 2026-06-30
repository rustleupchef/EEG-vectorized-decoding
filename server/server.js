const Mindwave = require('mindwave');
const http = require('http');
const fs = require('fs');
const path = require('path');

let latestMindwaveData = {
    eeg: null,
    time: null
};

const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost:3000');
    const pathname = url.pathname;
    const method = req.method;
    
    if (pathname === '/mindwave/data' && method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(latestMindwaveData));
    } else if (pathname === '/collect' && method === 'GET') {
        const filePath = path.join(__dirname, 'html', 'display.html');
        fs.readFile(filePath, 'utf8', (err, data) => {
            if (err) {
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ status: 'error', message: 'Failed to load page' }));
                return;
            }
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data);
        });
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
});

mw.connect('/dev/rfcomm0');