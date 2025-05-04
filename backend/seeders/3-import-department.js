'use strict';

const fs = require('fs');
const csv = require('csv-parser');

module.exports = {
    async up (queryInterface, Sequelize) {
        const departments = []

        return new Promise((resolve, reject) => {
        fs.createReadStream('./uploads/instacart/departments.csv')
            .pipe(csv())
            .on('data', (row) => {
                const sanitized_departmentname = row.department
                    .toLowerCase()
                    .replace(/\s+/g, '_')
                    .replace(/[^a-z0-9_]/g, '')

                departments.push({
                    department_id: parseInt(row.department_id, 10),
                    department_name: row.department,
                    department_img: `instacart_logo/${sanitized_departmentname}.svg`,
                    createdAt: new Date(),
                    updatedAt: new Date(),
                })
            })
            .on('end', async () => {
                try {
                    await queryInterface.bulkInsert('Departments', departments, {});
                    resolve();
                } catch (error) {
                    reject(error);
                }
            })
        })
    },

    async down (queryInterface, Sequelize) {
        await queryInterface.bulkDelete('Departments', null, {});
    }
}