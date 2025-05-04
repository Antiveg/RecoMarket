const { Order, OrderProducts, Department, Aisle, Product } = require('../models')
const path = require('path')
const fs = require('fs')
const formatDate  = require('../utilities/utility')

const getOrders = async (req, res, next) => {
    try {
        const orders = await Order.findAll({
            attributes: {
                exclude: ['updatedAt', 'createdAt']
            }
        })
        res.status(200).json({
            message: "successfully fetch order info...",
            orders: orders
        })
    }catch(err){
        next(err)
    }
}

const getUserOrders = async (req, res, next) => {
    try {
        const uid = +req.params.uid
        const userorders = await Order.findAll({
            attributes: {
                exclude: ['createdAt','updatedAt','eval_set','order_number','order_dow','order_hour_of_day']
            },
            include: [
                {
                    model: OrderProducts,
                    attributes: {
                        exclude: ['createdAt','updatedAt','product_id', 'add_to_cart_order']
                    },
                    include: [
                        {
                            model: Product,
                            attributes: {
                                exclude: ['createdAt','updatedAt','product_latent']
                            },
                            include: [
                                {
                                    model: Aisle,
                                    attributes: {
                                        exclude: ['createdAt','updatedAt','aisle_img']
                                    }
                                },
                                {
                                    model: Department,
                                    attributes: {
                                        exclude: ['createdAt','updatedAt']
                                    }
                                }
                            ]
                        }
                    ]
                }
            ],
            where: { user_id: uid }
        })
        // const finalUserOrders = userorders
        const finalUserOrders = await Promise.all(userorders.map(async (order) => {
            const finalOrderProducts = await Promise.all(order.OrderProducts.map(async (product) => {
                const { reordered, Product } = product
                const productJSON = Product.get({ plain: true })
                const { Department, Aisle, ...others } = productJSON
                return {
                    ...others,
                    reordered,
                    ...Department,
                    ...Aisle,
                    user_count: null
                }
            }))
            return {
                order_id: order.order_id,
                days_since_prior_order: 14,
                orderproducts: finalOrderProducts
            }
        }))
        res.status(200).json({
            message: "successfully fetch detailed userOrders info...",
            userOrders: finalUserOrders
        })
    }catch(err){
        next(err)
    }
}

module.exports = { getOrders, getUserOrders }