// import 'dart:io';

import 'package:flutter/material.dart';


Widget defaultButton({
  double width = double.infinity,
  Color background = Colors.blue,
  required Function function,
  required String text,
}) =>
    Container(
      width: width,
      height: 60.0,
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(50.0),
      ),

      child: MaterialButton(
        onPressed: ()
        {
          function();
        },
        child: Text(
          text.toUpperCase(),
          style:const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.w900,
          ),
        ),
      ),
    );


Widget defaultFormField({
  required TextEditingController controller,
  required TextInputType type,
  bool isPassword = false,
  Function? onSubmit,
  Function? onChange,
  required Function validate,
  required String lable,
  required IconData prefix,
  required String underLineText,
  IconData? suffix,
  Function? suffixPressed,

}
    )=>TextFormField(
  controller: controller,
  obscureText: isPassword,
  keyboardType: TextInputType.emailAddress,
  validator: (value){
    if(value.toString().isEmpty)
    {
      return underLineText;
    }
    return null;
  },
  onFieldSubmitted: (String value){
    onSubmit!(value);
  },
  onChanged: (String value){
    onChange!(value);
  },
  decoration: InputDecoration(
    labelText: lable,
    border: const OutlineInputBorder(),
    prefixIcon: Icon(
      prefix,
    ),
    suffixIcon: suffix!=null ?
    IconButton(
        onPressed: suffixPressed!() ,
        icon: Icon(
          suffix,
        )
    ): null,
  ),
);


void navigateTo(context, widget ) => Navigator.push(
  context,
  MaterialPageRoute(
    builder: (context)=> widget ,
  ),
);