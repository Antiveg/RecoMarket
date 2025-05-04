const { OrderProducts } = require('../models')
const path = require('path')
const fs = require('fs')
const formatDate  = require('../utilities/utility')

const getOrderProducts = async (req, res, next) => {
    try {
        const orderProducts = await OrderProducts.findAll({
            attributes: {
                exclude: ['updatedAt', 'createdAt']
            }
        })
        res.status(200).json({
            message: "successfully fetch orderProduct info...",
            orderProducts: orderProducts
        })
    }catch(err){
        next(err)
    }
}

module.exports = { getOrderProducts }