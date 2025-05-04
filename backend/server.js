require('dotenv').config()
const express = require('express')
const cors = require('cors')
const cookieParser = require('cookie-parser')
const errorHandler = require('./middlewares/errorHandling')

const UploadRouter = require('./routes/upload_routes')
const MainRouter = require('./routes/main_routes')
// const ApiRouter = require('./routes/api_routes')

const app = express()

app.use(cookieParser())
app.use(cors({
    origin: [process.env.AI_SERVER_DEV, process.env.FRONTEND_SERVER_DEV],
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,
}))

app.use(express.json())
app.use(express.urlencoded({ extended: true }))
app.use(UploadRouter)
app.use(MainRouter)
// app.use(ApiRouter)
app.use(errorHandler)

app.listen(process.env.PORT, () => {
    console.log(`Server is running on http://localhost:${process.env.PORT}`);
});