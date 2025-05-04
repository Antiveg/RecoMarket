'use strict'
const { Model } = require('sequelize')
module.exports = (sequelize, DataTypes) => {

    class Product extends Model {

        static associate(models){
            this.belongsTo(models.Aisle, { foreignKey: 'aisle_id' })
            this.belongsTo(models.Department, { foreignKey: 'department_id' })
            this.hasMany(models.OrderProducts, { foreignKey: 'product_id' })
        }

        // static async getUserCount(productId) {
        //     const UserCount = await sequelize.models.User.count({
        //         include: [
        //             {
        //                 model: sequelize.models.Order,
        //                 include: [
        //                     {
        //                         model: sequelize.models.OrderProducts,
        //                         where: { product_id: productId },
        //                         attributes: []
        //                     }
        //                 ],
        //                 attributes: [],
        //                 group: ['User.user_id']
        //             }
        //         ],
        //         distinct: true,
        //     })
        //     return UserCount
        // }

        static async getUserCount(productId) {
            const result = await sequelize.query(
                `
                SELECT 
                  COUNT(DISTINCT u."user_id") AS "user_count" 
                FROM "Products" p
                JOIN "OrderProducts" op ON op."product_id" = p."product_id"
                JOIN "Orders" o ON o."order_id" = op."order_id"
                JOIN "Users" u ON u."user_id" = o."user_id"
                WHERE p."product_id" = :product_id
                GROUP BY p."product_id"
                `,
                {
                  replacements: { product_id: productId },  // Passing the product_id to the query
                  type: sequelize.QueryTypes.SELECT, // Ensuring the result is returned as a select query
                }
            )
            return result && result[0] ? result[0].user_count : 0
        }
    }
    
    Product.init({
        product_id: {
            type: DataTypes.INTEGER,
            autoIncrement: true,
            primaryKey: true
        },
        product_name: {
            type: DataTypes.STRING,
            allowNull: false,
        },
        aisle_id: {
            type: DataTypes.INTEGER,
            allowNull: false,
            references: { model: 'Aisles', key: 'id' },
        },
        department_id: {
            type: DataTypes.INTEGER,
            allowNull: false,
            references: { model: 'Departments', key: 'id' },
        },
        product_latent: {
            type: DataTypes.ARRAY(DataTypes.FLOAT),
            allowNull: true, 
        },
        user_count: {
            type: DataTypes.INTEGER,
            allowNull: true,
        },
    }, {
        sequelize,
        timestamps: true,
        modelName: 'Product'
    })

    return Product
}