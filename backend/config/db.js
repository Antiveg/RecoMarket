const { Sequelize } = require('sequelize')

// insert configuration from .env
const config = require('./config')
const env = process.env.NODE_ENV || 'development'
const dbConfig = config[env]

// Create a new Sequelize instance to connect to your PostgreSQL database
const sequelize = new Sequelize({
    database: dbConfig.database,
    username: dbConfig.username,
    password: dbConfig.password,
    host: dbConfig.host,
    dialect: dbConfig.dialect,
    logging: false
})

// Connection Testing
(async () => {
    try {
        await sequelize.authenticate()
        console.log('Connection has been established successfully.');
    } catch (error) {
        console.error('Unable to connect to the database:', error)
    }
})()

module.exports = { sequelize }