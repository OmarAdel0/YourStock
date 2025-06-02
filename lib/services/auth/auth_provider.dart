import 'package:firebase_auth/firebase_auth.dart';
import 'package:yourstock/services/auth/auth_user.dart';

abstract class AuthProvider {
  Future<void> initialize();
  AuthUser? get currentUser;
  String? get userInfo;
  Future<AuthUser> logIn({
    required String email,
    required String password,
  });

  Future<AuthUser> createUser({
    required String email,
    required String password,
    required String confirmpassword,
  });

  Future<void> logOut();

  Future<User?> signInWithGoogle();

  Future<void> sendEmailVerification();
}
