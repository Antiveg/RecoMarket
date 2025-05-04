const { Aisle } = require('../models')
const path = require('path')
const fs = require('fs')
const formatDate  = require('../utilities/utility')

const getAisles = async (req, res, next) => {
    try {
        const aisles = await Aisle.findAll({
            attributes: {
                exclude: ['updatedAt', 'createdAt']
            }
        })
        res.status(200).json({
            message: "successfully fetch aisle info...",
            aisles: aisles
        })
    }catch(err){
        next(err)
    }
}

module.exports = { getAisles }