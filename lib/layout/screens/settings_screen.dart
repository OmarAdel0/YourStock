import 'package:flutter/material.dart';
import 'package:yourstock/constants/routes.dart';
import 'package:yourstock/services/auth/auth_exeptions.dart';
import 'package:yourstock/services/auth/auth_service.dart';
import 'package:yourstock/services/auth/firebase_auth_provider.dart';
import 'package:yourstock/services/crud/cloud_firestore_service.dart';
import 'package:yourstock/shared/cubit/cubit.dart';
import 'package:yourstock/utilities/show_error_dialog.dart';

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    FirebaseAuthProvider authProvider = FirebaseAuthProvider();
    final currentUser = authProvider.currentUser;
    return Scaffold(
        appBar: null, // Hide the app bar
        body: Center(
            child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Expanded(
              child: Padding(
                padding: const EdgeInsets.only(top: 50.0),
                child: Text(currentUser!.email.toString(),
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                    )),
              ),
              // Empty container to occupy the remaining space
            ),
            Align(
                alignment: Alignment.center,
                child: Column(
                  children: [
                    ElevatedButton(
                        onPressed: () async {
                          AppCubit.get(context);
                          final db = CloudDb();
                          final authProvider = FirebaseAuthProvider();
                          try {
                            db.deleteWatchlist();
                            authProvider.deleteUser();
                            Navigator.of(context).pushNamedAndRemoveUntil(
                              loginRoute,
                              (_) => false,
                            );
                          } on RequiresRecentLogin {
                            await showErrorDialog(
                              context,
                              'Requires Recent Login, Try to re-login and try again.',
                            );
                            await authProvider.logOut();
                            Navigator.of(context).pushNamedAndRemoveUntil(
                              loginRoute,
                              (_) => false,
                            );
                          } catch (e) {
                            if (e is GenericAuthException) {
                              await showErrorDialog(
                                context,
                                'Authentication error',
                              );
                            }
                          }
                        },
                        style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.red),
                        child: const Text("Delete this account")),
                    const SizedBox(
                      height: 5,
                    ),
                    ElevatedButton(
                        onPressed: () async {
                          final shouldLogout = await showLogoutDialog(context);
                          if (shouldLogout) {
                            await AuthService.firebase().logOut();
                            Navigator.of(context).pushNamedAndRemoveUntil(
                              loginRoute,
                              (_) => false,
                            );
                          }
                        },
                        style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.grey),
                        child: const Text("Logout")),
                  ],
                )),
            Expanded(
              child:
                  Container(), // Empty container to occupy the remaining space
            ),
          ],
        )));
  }
}

Future<bool> showLogoutDialog(context) {
  return showDialog<bool>(
    context: context,
    builder: (context) {
      return AlertDialog(
        title: const Text('log out'),
        content: const Text('Are you sure that you want to log out?'),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pop(false);
            },
            child: const Text('Cancle'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(context).pop(true);
            },
            child: const Text('Log out'),
          ),
        ],
      );
    },
  ).then((value) => value ?? false);
}
