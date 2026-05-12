const express = require('express');
const path = require('path');
const app = express();

// Serve static files
app.use(express.static(path.join(__dirname)));

// Serve index.html for root path
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Port from environment or default to 3000
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
  console.log(`AI Object Detection app listening on port ${PORT}`);
  console.log(`Open http://localhost:${PORT} in your browser`);
});
