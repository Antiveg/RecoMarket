'use strict';

module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.createTable('Orders', {
      order_id: {
        type: Sequelize.INTEGER,
        primaryKey: true,
        autoIncrement: true,
      },
      user_id: {
          type: Sequelize.INTEGER,
          allowNull: false,
          references: { model: 'Users', key: 'user_id' },
      },
      eval_set: {
          type: Sequelize.STRING,
          allowNull: false,
      },
      order_number: {
          type: Sequelize.INTEGER,
          allowNull: false,
      },
      order_dow: {
          type: Sequelize.INTEGER,
          allowNull: false,
      },
      order_hour_of_day: {
          type: Sequelize.INTEGER,
          allowNull: false,
      },
      days_since_prior_order: {
          type: Sequelize.FLOAT,
          defaultValue: 0
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
      }
    });
  },

  down: async (queryInterface, Sequelize) => {
    await queryInterface.dropTable('Orders');
  }
};
