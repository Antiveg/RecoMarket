'use strict'
const { Model } = require('sequelize')
module.exports = (sequelize, DataTypes) => {

    class Order extends Model {

        static associate(models){
            this.belongsTo(models.User, { foreignKey: 'user_id' })
            this.hasMany(models.OrderProducts, { foreignKey: 'order_id' })
        }

    }

    Order.init({
        order_id: {
            type: DataTypes.INTEGER,
            autoIncrement: true,
            primaryKey: true
        },
        user_id: {
            type: DataTypes.INTEGER,
            allowNull: false,
            references: { model: 'Users', key: 'user_id' },
        },
        eval_set: {
            type: DataTypes.STRING,
            allowNull: false,
        },
        order_number: {
            type: DataTypes.INTEGER,
            allowNull: false,
        },
        order_dow: {
            type: DataTypes.INTEGER,
            allowNull: false,
        },
        order_hour_of_day: {
            type: DataTypes.INTEGER,
            allowNull: false,
        },
        days_since_prior_order: {
            type: DataTypes.FLOAT,
            allowNull: true,
        },
    }, {
        sequelize,
        timestamps: true,
        modelName: 'Order'
    })

    return Order
}