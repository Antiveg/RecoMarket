'use strict'
const { Model } = require('sequelize')
module.exports = (sequelize, DataTypes) => {

    class User extends Model {

        static associate(models){
            this.hasMany(models.Order, { foreignKey: 'user_id' })
        }

        static async getOrderCount(userId) {
            const orderCount = await sequelize.models.Order.count({
                attributes: [],
                where: { user_id: userId },
                distinct: true,
            })
            return orderCount
        }
    }
    
    User.init({
        user_id: {
            type: DataTypes.INTEGER,
            autoIncrement: true,
            primaryKey: true
        },
        user_name: {
            type: DataTypes.STRING,
            allowNull: false,
            defaultValue: 'john doe'
        },
        user_img: {
            type: DataTypes.STRING,
            allowNull: false,
            defaultValue: 'userprofiles/consumer_temp001.svg'
        },
        user_latent: {
            type: DataTypes.ARRAY(DataTypes.FLOAT),
            allowNull: true, 
        },
        order_count: {
            type: DataTypes.INTEGER,
            allowNull: true,
        },
    }, {
        sequelize,
        timestamps: true,
        modelName: 'User'
    })

    return User
}