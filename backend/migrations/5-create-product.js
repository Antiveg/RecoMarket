'use strict';

module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.createTable('Products', {
      product_id: {
        type: Sequelize.INTEGER,
        primaryKey: true,
        autoIncrement: true,
      },
      product_name: {
          type: Sequelize.STRING,
          allowNull: false,
      },
      aisle_id: {
          type: Sequelize.INTEGER,
          allowNull: false,
          references: { model: 'Aisles', key: 'aisle_id' },
      },
      department_id: {
          type: Sequelize.INTEGER,
          allowNull: false,
          references: { model: 'Departments', key: 'department_id' },
      },
      product_latent: {
        type: Sequelize.ARRAY(Sequelize.FLOAT),
        allowNull: true, 
      },
      user_count: {
        type: Sequelize.INTEGER,
        allowNull: true,
      },
      createdAt: {
        type: Sequelize.DATE,
        allowNull: false,
        defaultValue: Sequelize.NOW,
      },
      updatedAt: {
        type: Sequelize.DATE,
        allowNull: false,
        defaultValue: Sequelize.NOW,
      },
    });
  },

  down: async (queryInterface, Sequelize) => {
    await queryInterface.dropTable('Products');
  }
};
