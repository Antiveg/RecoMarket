'use strict'

const fs = require('fs')
const csv = require('csv-parser')

module.exports = {
    async up (queryInterface, Sequelize) {
        const products = []

        return new Promise((resolve, reject) => {
        fs.createReadStream('./uploads/instacart/products.csv')
            .pipe(csv())
            .on('data', (row) => {

                products.push({
                    product_id: parseInt(row.product_id, 10),
                    product_name: row.product_name,
                    aisle_id: parseInt(row.aisle_id),
                    department_id: parseInt(row.department_id),
                    product_latent: null,
                    user_count: null,
                    createdAt: new Date(),
                    updatedAt: new Date(),
                })
            })
            .on('end', async () => {
                try {
                    await queryInterface.bulkInsert('Products', products, {})
                    resolve()
                } catch (error) {
                    reject(error)
                }
            })
        })
    },

    async down (queryInterface, Sequelize) {
        await queryInterface.bulkDelete('Products', null, {});
    }
}