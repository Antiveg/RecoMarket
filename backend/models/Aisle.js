'use strict'
const { Model } = require('sequelize')
module.exports = (sequelize, DataTypes) => {

    class Aisle extends Model {

        static associate(models){
            this.hasMany(models.Product, { foreignKey: 'aisle_id' })
        }

    }
    
    Aisle.init({
        aisle_id: {
            type: DataTypes.INTEGER,
            primaryKey: true,
            autoIncrement: true,
        },
        aisle_name: {
            type: DataTypes.STRING,
            allowNull: false,
        },
        aisle_img: {
            type: DataTypes.STRING,
            allowNull: true,
            defaultValue: null
        }
    }, {
        sequelize,
        timestamps: true,
        modelName: 'Aisle'
    })

    return Aisle
}