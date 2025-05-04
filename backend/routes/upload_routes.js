const express = require('express')
const app = express()
const router = express.Router()
const path = require('path')

router.use('/uploads/userprofiles', express.static(path.join(__dirname, '../uploads/userprofiles')))
router.use('/uploads/instacart_logo', express.static(path.join(__dirname, '../uploads/instacart_logo')))
router.use('/uploads/instacart', (req, res) => {
    res.status(403).send('Forbidden: This folder is restricted.')
})

module.exports = router

// PREVIOUS EXAMPLE
// app.use('/uploads', (req, res, next) => {
//     res.header('Access-Control-Allow-Origin', 'http://localhost:5173');
//     res.header('Access-Control-Allow-Methods', 'GET, HEAD, OPTIONS');
//     res.header('Access-Control-Allow-Headers', 'Content-Type');
//     next();
// }, express.static(path.join(__dirname, 'uploads')));