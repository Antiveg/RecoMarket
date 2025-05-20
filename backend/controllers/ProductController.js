const { sequelize, Product, Department, Aisle, Order, OrderProducts } = require('../models')
const path = require('path')
const axios = require('axios')
const fs = require('fs')
const formatDate  = require('../utilities/utility')

const getProducts = async (req, res, next) => {
    try {
        const products = await Product.findAll({
            attributes: {
                exclude: ['updatedAt', 'createdAt']
            }
        })
        res.status(200).json({
            message: "successfully fetch Product info...",
            products: products
        })
    }catch(err){
        next(err)
    }
}

const getDetailedProducts = async (req, res, next) => {
    try {
        const products = await Product.findAll({
            attributes: {
                exclude: ['updatedAt', 'createdAt']
            },
            include: [
                {
                    model: Department,
                    attributes: {
                        exclude: ['createdAt','updatedAt']
                    },
                },
                {
                    model: Aisle,
                    attributes: {
                        exclude: ['createdAt','updatedAt']
                    },
                },
            ],
            limit: 1000
        })
        const updates = products.map(async (product) => {
            if (product.user_count == null) {
                const userCount = await Product.getUserCount(product.product_id)
                await Product.update(
                    { user_count: userCount }, 
                    { where: { product_id: product.product_id } }
                )
                product.user_count = userCount
            }
        })
        await Promise.all(updates)
        const finalProducts = []
        for (const product of products) {
            const productJSON = product.get({ plain: true })
            const { Department, Aisle, ...others } = productJSON
            finalProducts.push({
                ...others,
                ...Department,
                ...Aisle,
            })
        }
        res.status(200).json({
            message: "successfully fetch Product info...",
            products: finalProducts
        })
    }catch(err){
        next(err)
    }
}

const getUserProducts = async (req, res, next) => {
    try {
        const uid = +req.params.uid
        const userproducts = await Order.findAll({
            where: { user_id: uid },
            attributes: [],
            include: [
                {
                    model: OrderProducts,
                    attributes: ['product_id'],
                    include: [
                        {
                            model: Product,
                            attributes: ['product_id','product_name'],
                            include: [
                                {
                                    model: Department,
                                    attributes: ['department_id', 'department_name', 'department_img']
                                },
                                {
                                    model: Aisle,
                                    attributes: ['aisle_id', 'aisle_name']
                                }
                            ],
                        }
                    ]
                }
            ]
        })
        const productCountMap = {}
        userproducts.forEach(order => {
            order.OrderProducts.forEach(orderProduct => {
                const product = orderProduct.Product.get({ plain: true })
                const productId = orderProduct.product_id
                if(!productCountMap[productId]){
                    const { Department, Aisle, ...others } = product
                    productCountMap[productId] = {
                        ...others,
                        ...Department,
                        ...Aisle,
                        purchase_count: 0
                    }
                }
                productCountMap[productId].purchase_count += 1
            })
        })
        const finalUserProducts = Object.values(productCountMap)
        res.status(200).json({
            message: "successfully fetch userProduct info...",
            userProducts: finalUserProducts
        })
    }catch(err){
        next(err)
    }
}

const getUserRecommendations = async (req, res, next) => {
    try {
        const uid = +req.params.uid
        const { collabweight, contentweight } = req.body
        const response = await axios.get(`${process.env.AI_SERVER_DEV}/recommendations/${uid}`, {
            params: {
                collabweight, contentweight
            }
        })
        const recommendations = response.data.recommendations
        const score_map = new Map(recommendations.map(([id, score]) => [id, score]))
        const recommended_ids = recommendations.map(([id]) => id)

        const products = await Product.findAll({
            attributes: {
                exclude: ['createdAt','updatedAt']
            },
            include: [
                {
                    model: Aisle,
                    attributes: ['aisle_name']
                },
                {
                    model: Department,
                    attributes: ['department_name','department_img']
                }
            ],
            where: { product_id: recommended_ids }
        })
        const sortedProducts = recommended_ids.map((id) => {
            return products.find(product => product.product_id === id)
        })
        const finalProducts = sortedProducts.map((product) => {
            const productData = product.get({ plain: true })
            const { Aisle, Department, ...others } = productData
            return {
                ...others,
                ...Aisle,
                ...Department,
                score: score_map.get(product.product_id)
            }
        })
        res.status(200).json({
            message: 'successfully get user recommendations',
            recommendations: finalProducts
        })
    }catch(error){
        next(error)
    }
}

module.exports = { getProducts, getDetailedProducts, getUserProducts, getUserRecommendations }