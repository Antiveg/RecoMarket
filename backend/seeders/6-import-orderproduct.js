'use strict'

const fs = require('fs')
const csv = require('csv-parser')

module.exports = {
    async up (queryInterface, Sequelize) {
        const orderproducts = []

        return new Promise((resolve, reject) => {
        fs.createReadStream('./uploads/instacart/order_products__train.csv')
            .pipe(csv())
            .on('data', (row) => {

                orderproducts.push({
                    order_id: parseInt(row.order_id, 10),
                    product_id: parseInt(row.product_id, 10),
                    add_to_cart_order: parseInt(row.add_to_cart_order, 10),
                    reordered: Boolean(parseInt(row.reordered, 10)),
                    createdAt: new Date(),
                    updatedAt: new Date(),
                })
            })
            .on('end', async () => {
                try {
                    await queryInterface.bulkInsert('OrderProducts', orderproducts, {})
                    resolve()
                } catch (error) {
                    reject(error)
                }
            })
        })
    },

    async down (queryInterface, Sequelize) {
        await queryInterface.bulkDelete('OrderProducts', null, {});
    }
}