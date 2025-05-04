const { User } = require('../models')
const path = require('path')
const fs = require('fs')
const formatDate  = require('../utilities/utility')

const getUsers = async (req, res, next) => {
    try {
        const users = await User.findAll({
            attributes: {
                exclude: ['updatedAt', 'createdAt']
            },
            limit: 1000
        })
        res.status(200).json({
            message: "successfully fetch User info...",
            users: users
        })
    }catch(err){
        next(err)
    }
}

const getDetailedUsers = async (req, res, next) => {
    try {
        const users = await User.findAll({
            attributes: {
                exclude: ['updatedAt', 'createdAt']
            },
            limit: 1000
        })
        const updates = users.map(async (user) => {
            if (user.order_count == null) {
                const orderCount = await User.getOrderCount(user.user_id)
                await User.update(
                    { order_count: orderCount }, 
                    { where: { user_id: user.user_id } }
                )
                user.order_count = orderCount
            }
        })
        await Promise.all(updates)
        const plainUsers = users.map(user => user.get({ plain: true }));
        res.status(200).json({
            message: "Successfully fetched user info...",
            users: plainUsers
        })
    } catch (err) {
        next(err)
    }
}

module.exports = { getUsers, getDetailedUsers }