// const express = require(express)
const mongoose = require("mongoose")
const existingUserSchema = new mongoose.Schema({
    userId : {
        type : String 
    },
    dwellTime :{
        type : String
    },
    flightTime :{
        type : String
    },
    interKeyTime :{
        type : String
    },
})
const existingUserModel = mongoose.model("existingUser",existingUserSchema)

module.exports = existingUserModel