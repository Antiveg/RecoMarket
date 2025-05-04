'use strict';

module.exports = {
  up: async (queryInterface, Sequelize) => {
    await queryInterface.createTable('Users', {
      user_id: {
        type: Sequelize.INTEGER,
        primaryKey: true,
        autoIncrement: true
      },
      user_name: {
        type: Sequelize.STRING,
        allowNull: false,
        defaultValue: 'john doe'
      },
      user_img: {
        type: Sequelize.STRING,
        allowNull: false, 
      },
      user_latent: {
        type: Sequelize.ARRAY(Sequelize.FLOAT),
        allowNull: true, 
      },
      order_count: {
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
      }
    });
  },

  down: async (queryInterface, Sequelize) => {
    await queryInterface.dropTable('Users');
  }
};
