'use strict'
const { Model } = require('sequelize')
module.exports = (sequelize, DataTypes) => {

    class OrderProducts extends Model {

        static associate(models){
            this.belongsTo(models.Order, { foreignKey: 'order_id' })
            this.belongsTo(models.Product, { foreignKey: 'product_id' })
        }

    }
    
    OrderProducts.init({
        order_id: {
            type: DataTypes.INTEGER,
            allowNull: false,
            primaryKey: true,
            references: { model: 'Orders', key: 'order_id' },
        },
        product_id: {
            type: DataTypes.INTEGER,
            allowNull: false,
            primaryKey: true,
            references: { model: 'Products', key: 'product_id' },
        },
        add_to_cart_order: {
            type: DataTypes.INTEGER,
            allowNull: false,
        },
        reordered: {
            type: DataTypes.BOOLEAN,
            allowNull: false
        }
    }, {
        sequelize,
        timestamps: true,
        modelName: 'OrderProducts'
    })

    return OrderProducts
}