// const express = require(express)
const mongoose = require("mongoose")
const UserSchema = new mongoose.Schema({
    email : {
        type : String
    },
    password :{
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
const userModel = mongoose.model("user",UserSchema)

module.exports = userModel