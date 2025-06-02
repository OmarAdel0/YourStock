// import 'dart:developer';

import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/foundation.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:yourstock/firebase_options.dart';
import 'package:yourstock/services/auth/auth_user.dart';
import 'package:yourstock/services/auth/auth_provider.dart';
import 'package:yourstock/services/auth/auth_exeptions.dart';
import 'package:firebase_auth/firebase_auth.dart'
    show
        AuthCredential,
        FirebaseAuth,
        FirebaseAuthException,
        GoogleAuthProvider,
        User,
        UserCredential;
import 'package:yourstock/services/crud/cloud_firestore_service.dart';

class FirebaseAuthProvider implements AuthProvider {
  GoogleSignIn? initializeGoogleSignin() {
    if (!kIsWeb) {
      final GoogleSignIn googleSignIn = GoogleSignIn();
      return googleSignIn;
    } else {
      return null;
    }
  }

  @override
  Future<AuthUser> createUser({
    required String email,
    required String password,
    required String confirmpassword,
  }) async {
    if (password == confirmpassword) {
      try {
        await FirebaseAuth.instance.createUserWithEmailAndPassword(
          email: email,
          password: password,
        );
        final user = currentUser;
        if (user != null) {
          final watchlistDb = CloudDb();
          watchlistDb.createWatchlist();
          return user;
        } else {
          throw UserNotLoggedInException();
        }
      } on FirebaseAuthException catch (e) {
        if (e.code == 'weak-password') {
          throw WeakPasswordAuthException();
        } else if (e.code == 'email-already-in-use') {
          throw EmailAlreadyInUseAuthException();
        } else if (e.code == 'invalid-email') {
          throw InvalidEmailAuthException();
        } else if (e.code == 'unknown') {
          throw GenericAuthException();
        } else {
          throw GenericAuthException();
        }
      } catch (_) {
        throw GenericAuthException();
      }
    } else {
      throw DifferentConfirmPassword();
    }
  }

  @override
  AuthUser? get currentUser {
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      return AuthUser.fromFirebase(user);
    } else {
      return null;
    }
  }

  @override
  Future<AuthUser> logIn({
    required String email,
    required String password,
  }) async {
    try {
      await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
      final user = currentUser;
      if (user != null) {
        final watchlistDb = CloudDb();
        watchlistDb.createWatchlist();
        return user;
      } else {
        throw UserNotLoggedInException();
      }
    } on FirebaseAuthException catch (e) {
      if (e.code == 'user-not-found') {
        throw UserNotFoundAuthException();
      } else if (e.code == 'wrong-password') {
        throw WrongPasswordAuthException();
      } else if (e.code == 'invalid-email') {
        throw InvalidEmailAuthException();
      } else {
        throw GenericAuthException();
      }
    } catch (_) {
      throw GenericAuthException();
    }
  }

  @override
  Future<void> logOut() async {
    final user = FirebaseAuth.instance.currentUser;
    GoogleSignIn? googleSignIn = initializeGoogleSignin();
    if (googleSignIn != null) {
      if (await googleSignIn.isSignedIn()) {
        await googleSignIn.disconnect();
      }
    }
    if (user != null) {
      await FirebaseAuth.instance.signOut();
    } else {
      throw UserNotLoggedInException();
    }
  }

  @override
  Future<void> sendEmailVerification() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      await user.sendEmailVerification();
    } else {
      throw UserNotLoggedInException();
    }
  }

  @override
  Future<User?> signInWithGoogle() async {
    GoogleSignIn? googleSignIn = initializeGoogleSignin();
    if (!kIsWeb) {
      FirebaseAuth auth = FirebaseAuth.instance;
      User? user;

      GoogleSignInAccount? googleSignInAccount = await googleSignIn!.signIn();

      if (googleSignInAccount != null) {
        final GoogleSignInAuthentication googleSignInAuthentication =
            await googleSignInAccount.authentication;

        final AuthCredential credential = GoogleAuthProvider.credential(
          accessToken: googleSignInAuthentication.accessToken,
          idToken: googleSignInAuthentication.idToken,
        );

        try {
          final UserCredential userCredential =
              await auth.signInWithCredential(credential);

          user = userCredential.user;
        } on FirebaseAuthException catch (e) {
          if (e.code == 'account-exists-with-different-credential') {
            throw AccountExistsWithDifferentCredentialAuthException();
          } else if (e.code == 'invalid-credential') {
            throw InvalidCredentialAuthException();
          }
        } catch (e) {
          throw GenericAuthException();
        }
      }

      return user;
    }
    return null;
  }

  Future<void> deleteUser() async {
    try {
      await FirebaseAuth.instance.currentUser!.delete();
      if (!kIsWeb) {
        GoogleSignIn? googleSignIn = initializeGoogleSignin();
        if (await googleSignIn!.isSignedIn()) {
          await googleSignIn.disconnect();
        }
      }
    } on FirebaseAuthException catch (e) {
      if (e.code == "requires-recent-login") {
        throw RequiresRecentLogin();
      } else {
        throw GenericAuthException();
      }
    }
  }

  @override
  Future<void> initialize() async {
    if (Firebase.apps.isEmpty) {
      await Firebase.initializeApp(
        options: DefaultFirebaseOptions.currentPlatform,
      );
    }
  }

  @override
  String? get userInfo => FirebaseAuth.instance.currentUser?.email;
}
