'use strict'

const fs = require('fs')
const csv = require('csv-parser')

module.exports = {
    async up (queryInterface, Sequelize) {
        const orders = []

        return new Promise((resolve, reject) => {
        fs.createReadStream('./uploads/instacart/orders.csv')
            .pipe(csv())
            .on('data', (row) => {
                if(row.eval_set == 'train'){
                    orders.push({
                        order_id: parseInt(row.order_id, 10),
                        user_id: parseInt(row.user_id, 10),
                        eval_set: row.eval_set,
                        order_number: parseInt(row.order_number, 10),
                        order_dow: parseInt(row.order_dow, 10),
                        order_hour_of_day: parseInt(row.order_hour_of_day, 10),
                        days_since_prior_order: (row.days_since_prior_order ? parseFloat(row.days_since_prior_order) : null),
                        createdAt: new Date(),
                        updatedAt: new Date(),
                    })
                }
            })
            .on('end', async () => {
                try {
                    await queryInterface.bulkInsert('Orders', orders, {})
                    resolve()
                } catch (error) {
                    reject(error)
                }
            })
        })
    },

    async down (queryInterface, Sequelize) {
        await queryInterface.bulkDelete('Orders', null, {});
    }
}