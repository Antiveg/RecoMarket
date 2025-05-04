const { Department } = require('../models')
const path = require('path')
const fs = require('fs')
const formatDate  = require('../utilities/utility')

const getDepartments = async (req, res, next) => {
    try {
        const departments = await Department.findAll({
            attributes: {
                exclude: ['updatedAt', 'createdAt']
            }
        })
        res.status(200).json({
            message: "successfully fetch department info...",
            departments: departments
        })
    }catch(err){
        next(err)
    }
}

module.exports = { getDepartments }