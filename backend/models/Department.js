'use strict'
const { Model } = require('sequelize')
module.exports = (sequelize, DataTypes) => {

    class Department extends Model {

        static associate(models){
            this.hasMany(models.Product, { foreignKey: 'department_id' })
        }
    }
    
    Department.init({
        department_id: {
            type: DataTypes.INTEGER,
            primaryKey: true,
            autoIncrement: true,
        },
        department_name: {
            type: DataTypes.STRING,
            allowNull: false,
        },
        department_img: {
            type: DataTypes.STRING,
            allowNull: true
        },
    }, {
        sequelize,
        timestamps: true,
        modelName: 'Department'
    })

    return Department
}