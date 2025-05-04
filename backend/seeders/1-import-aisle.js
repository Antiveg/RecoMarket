'use strict';

const fs = require('fs');
const csv = require('csv-parser');

module.exports = {
    async up (queryInterface, Sequelize) {
        const aisles = []

        return new Promise((resolve, reject) => {
        fs.createReadStream('./uploads/instacart/aisles.csv')
            .pipe(csv())
            .on('data', (row) => {
                // const sanitized_aislename = row.aisle
                //     .toLowerCase()
                //     .replace(/\s+/g, '_')
                //     .replace(/[^a-z0-9_]/g, '')

                aisles.push({
                    aisle_id: parseInt(row.aisle_id, 10),
                    aisle_name: row.aisle,
                    // aisle_img: `instacart_logo/${sanitized_aislename}.svg`,
                    createdAt: new Date(),
                    updatedAt: new Date(),
                })
            })
            .on('end', async () => {
                try {
                    await queryInterface.bulkInsert('Aisles', aisles, {});
                    resolve();
                } catch (error) {
                    reject(error);
                }
            })
        })
    },

    async down (queryInterface, Sequelize) {
        await queryInterface.bulkDelete('Aisles', null, {});
    }
}