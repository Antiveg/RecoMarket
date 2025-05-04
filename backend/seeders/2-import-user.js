'use strict'

const fs = require('fs')
const csv = require('csv-parser')
const faker = require('faker') // or '@faker-js/faker'

module.exports = {
    async up (queryInterface, Sequelize) {
        const userIds = new Set()
        const users = []

        return new Promise((resolve, reject) => {
        fs.createReadStream('./uploads/instacart/orders.csv')
            .pipe(csv())
            .on('data', (row) => {
                const userId = parseInt(row.user_id, 10)
                if (row.eval_set == 'train' && !userIds.has(userId)) {
                    userIds.add(userId)
                    users.push({
                        user_id: userId,
                        user_name: faker.name.findName(),
                        user_latent: null,
                        user_img: 'userprofiles/customer_temp001.svg',
                        order_count: null,
                        createdAt: new Date(),
                        updatedAt: new Date(),
                    })
                }
            })
            .on('end', async () => {
                try {
                    await queryInterface.bulkInsert('Users', users, {})
                    resolve()
                } catch (error) {
                    reject(error)
                }
            })
        })
    },

    async down (queryInterface, Sequelize) {
        await queryInterface.bulkDelete('Users', null, {})
    }
}
