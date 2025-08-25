const express = require("express")
const userModel = require("../Models/existingUserModel")
const router = express.Router()
router.post("/",async (req,res)=>{
    // console.log(req.body)
    const {email,password} = req.body.formData

    const Users = await userModel.find()
    // console.log("Users stored : ",Users)

    Users.forEach(storedUser => {
        if(storedUser.email === email){
            res.status(200).json({message :"User Already Exists",out : true})
            return
        }
    });
    const newUser = new userModel({
        email : email,
        password : password
    })
    await newUser.save()
    res.status(200).json({message:"data recieved success",out:false})
})
module.exports = router