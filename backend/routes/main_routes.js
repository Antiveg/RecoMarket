const express = require('express')
const app = express()
const router = express.Router()
const path = require('path')

const errorHandler = require('../middlewares/errorHandling')

const { getProducts, getDetailedProducts, getUserProducts, getUserRecommendations } = require('../controllers/ProductController')
const { getUsers, getDetailedUsers } = require('../controllers/UserController')
const { getOrders, getUserOrders } = require('../controllers/OrderController')

router.get('/products', getProducts)
router.get('/detailed-products', getDetailedProducts)
router.get('/users', getUsers)
router.get('/detailed-users', getDetailedUsers)
router.get('/user-orders/:uid', getUserOrders)
router.get('/user-products/:uid', getUserProducts)
router.post('/recommendations/:uid', getUserRecommendations)

router.use(errorHandler)

module.exports = router