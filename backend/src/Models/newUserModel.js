// const { mongo } = require("mongoose")
const mongoose = require("moongoose")
const newUserSchema = new mongoose.Schema({
    email : {
        type : String
    },
    password : {
        type : String
    }
})

module.exports = mongoose.model("newUser",newUserSchema)