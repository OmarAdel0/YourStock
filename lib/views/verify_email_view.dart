// ignore_for_file: use_build_context_synchronously

import 'package:flutter/material.dart';
import 'package:yourstock/constants/routes.dart';
import 'package:yourstock/services/auth/auth_service.dart';

class VerifyEmailView extends StatefulWidget {
  const VerifyEmailView({super.key});

  @override
  State<VerifyEmailView> createState() => _VerifyEmailViewState();
}

class _VerifyEmailViewState extends State<VerifyEmailView> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Verify account'),
      ),
      body: Column(
        children: [
          const Text("We've already sent you email verification."),
          const Text("If you didn't recived an email verification."),
          TextButton(
            onPressed: () async {
              //get the logged in/registered user, then send verification email.
              await AuthService.firebase().sendEmailVerification();
            },
            child: const Text('Send email verification, again.'),
          ),
          TextButton(
            onPressed: () async {
              await AuthService.firebase().logOut();
              Navigator.of(context).pushNamedAndRemoveUntil(
                registerRoute,
                (route) => false,
              );
            },
            child: const Text('Go back.'),
          ),
        ],
      ),
    );
  }
}
