import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:yourstock/constants/routes.dart';
import 'package:yourstock/services/auth/auth_exeptions.dart';
import 'package:yourstock/services/auth/auth_service.dart';
import 'package:yourstock/services/auth/google_signin_button.dart';
import 'package:yourstock/services/crud/cloud_firestore_service.dart';
import 'package:yourstock/utilities/show_error_dialog.dart';

class LoginView extends StatefulWidget {
  const LoginView({Key? key}) : super(key: key);

  @override
  State<LoginView> createState() => _LoginViewState();
}

class _LoginViewState extends State<LoginView> {
  late final TextEditingController _email;
  late final TextEditingController _password;

//connect the objects with the controllers witch text fields connected to.
  @override
  void initState() {
    _email = TextEditingController();
    _password = TextEditingController();
    super.initState();
  }

//disconnect the objects and the controllers.
  @override
  void dispose() {
    _email.dispose();
    _password.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Login'),
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
                hintText: "Enter your email here",
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
                hintText: "Enter your password here",
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(10),
            child: ElevatedButton(
              onPressed: () async {
                final email = _email.text;
                final password = _password.text;
                try {
                  await AuthService.firebase().logIn(
                    email: email,
                    password: password,
                  );
                  final user = AuthService.firebase().currentUser;
                  if (user?.isEmailVerified ?? false) {
                    final watchlistDb = CloudDb();
                    watchlistDb.createWatchlist();
                    Navigator.of(context).pushNamedAndRemoveUntil(
                      homeRoute,
                      (route) => false,
                    );
                  } else {
                    Navigator.of(context).pushNamedAndRemoveUntil(
                      verifyRoute,
                      (route) => false,
                    );
                  }
                } on UserNotFoundAuthException {
                  await showErrorDialog(
                    context,
                    'User not found',
                  );
                } on WrongPasswordAuthException {
                  await showErrorDialog(
                    context,
                    'wrong-password',
                  );
                } on InvalidEmailAuthException {
                  await showErrorDialog(
                    context,
                    'Invalid email',
                  );
                } on GenericAuthException {
                  await showErrorDialog(
                    context,
                    'Authentication error',
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
              child: const Text(
                "Login",
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.only(top: 5, bottom: 20),
            child: ElevatedButton(
                onPressed: (() {
                  //go to registration view.
                  Navigator.of(context).pushNamedAndRemoveUntil(
                    registerRoute,
                    (route) => false,
                  );
                }),
                style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: Colors.black),
                child: const Text("Don't have an account yet? signup here. ")),
          ),
          if (!kIsWeb)
            FutureBuilder(
              future: AuthService.firebase().initialize(),
              builder: (context, snapshot) {
                if (snapshot.hasError) {
                  asyncShowErrorDialog(context, 'Error initializing Firebase');
                } else if (snapshot.connectionState == ConnectionState.done) {
                  return const GoogleSignInButton();
                }
                return const CircularProgressIndicator(
                  valueColor: AlwaysStoppedAnimation<Color>(
                    Colors.orange,
                  ),
                );
              },
            ),
        ],
      ),
    );
  }
}
