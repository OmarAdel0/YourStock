// ignore_for_file: use_build_context_synchronously

import 'package:flutter/material.dart';
import 'package:yourstock/constants/routes.dart';
import 'package:yourstock/services/auth/auth_exeptions.dart';
import 'package:yourstock/services/auth/auth_service.dart';
import 'package:yourstock/utilities/show_error_dialog.dart';

class RegisterView extends StatefulWidget {
  const RegisterView({Key? key}) : super(key: key);

  @override
  State<RegisterView> createState() => _RegisterViewState();
}

class _RegisterViewState extends State<RegisterView> {
  late final TextEditingController _email;
  late final TextEditingController _password;
  late final TextEditingController _confirmpassword;

//connect the objects with the controllers witch text fields connected to.
  @override
  void initState() {
    _email = TextEditingController();
    _password = TextEditingController();
    _confirmpassword = TextEditingController();
    super.initState();
  }

//disconnect the objects and the controllers.
  @override
  void dispose() {
    _email.dispose();
    _password.dispose();
    _confirmpassword.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Signup"),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20.0, vertical: 10),
            child: TextField(
              controller: _email,
              enableSuggestions: false,
              autocorrect: false,
              keyboardType: TextInputType.emailAddress,
              decoration: const InputDecoration(
                hintText: "Ex:.. yourstock@example.com",
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20.0, vertical: 10),
            child: TextField(
              controller: _password,
              obscureText: true,
              enableSuggestions: false,
              autocorrect: false,
              decoration: const InputDecoration(
                hintText: "Enter Password here",
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20.0, vertical: 10),
            child: TextField(
              controller: _confirmpassword,
              obscureText: true,
              enableSuggestions: false,
              autocorrect: false,
              decoration: const InputDecoration(
                hintText: "Confirm password here",
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(10),
            child: ElevatedButton(
              onPressed: () async {
                final email = _email.text;
                final password = _password.text;
                final confirmpassword = _confirmpassword.text;
                try {
                  await AuthService.firebase().createUser(
                    email: email,
                    password: password,
                    confirmpassword: confirmpassword,
                  );
                  AuthService.firebase().sendEmailVerification();
                  Navigator.of(context).pushNamed(
                    verifyRoute,
                  );
                } on WeakPasswordAuthException {
                  await showErrorDialog(
                    context,
                    'Weak password.',
                  );
                } on EmailAlreadyInUseAuthException {
                  await showErrorDialog(
                    context,
                    'Email already in use.',
                  );
                } on InvalidEmailAuthException {
                  await showErrorDialog(
                    context,
                    'Invalid email.',
                  );
                } on GenericAuthException {
                  await showErrorDialog(
                    context,
                    'Faild to signup',
                  );
                } on DifferentConfirmPassword {
                  await showErrorDialog(
                    context,
                    'Password and Confirm Password are different',
                  );
                }
              },
              style: ElevatedButton.styleFrom(
                padding:
                    const EdgeInsets.symmetric(horizontal: 30, vertical: 13),
                textStyle: const TextStyle(
                  fontSize: 16,
                  color: Colors.white,
                ),
                alignment: Alignment.center,
                backgroundColor: Colors.deepOrange.withOpacity(0.7),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: const Text("Signup"),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(5),
            child: ElevatedButton(
              onPressed: () {
                Navigator.of(context).pushNamedAndRemoveUntil(
                  loginRoute,
                  (route) => false,
                );
              },
              style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white, foregroundColor: Colors.black),
              child: const Text("Already have an account? login here."),
            ),
          ),
        ],
      ),
    );
  }
}
