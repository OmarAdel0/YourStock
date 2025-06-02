import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/services.dart';
import 'package:yourstock/layout/splash_screen.dart';
import 'package:yourstock/shared/bloc_observer.dart';
import 'package:yourstock/shared/network/dio_helper.dart';
import 'firebase_options.dart';
import 'package:yourstock/services/auth/auth_service.dart';
import 'package:yourstock/views/login_view.dart';
import 'package:yourstock/layout/home_layout.dart';
import 'package:yourstock/views/register_view.dart';
import 'package:yourstock/views/verify_email_view.dart';
import 'package:yourstock/constants/routes.dart';

import 'package:flutter_bloc/flutter_bloc.dart';

MaterialColor primarySwatch = const MaterialColor(
  0xff0e1117,
  <int, Color>{
    50: Color(0xffe8eaf0),
    100: Color(0xffc5cbe9),
    200: Color(0xff9fa8da),
    300: Color(0xff7985cb),
    400: Color(0xff5c6bc0),
    500: Color(0xff3f51b5),
    600: Color(0xff394aae),
    700: Color(0xff3141a5),
    800: Color(0xff29399d),
    900: Color(0xff1b2d86),
  },
);

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  Bloc.observer = MyBlocObserver();
  DioHelper.init();
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false,
    title: 'Your Stock',
    theme: ThemeData(
      primarySwatch: primarySwatch,
      scaffoldBackgroundColor: Colors.white,
      appBarTheme: const AppBarTheme(
        iconTheme: IconThemeData(
          color: Colors.black,
        ),
        actionsIconTheme: IconThemeData(
          color: Colors.black,
        ),
        systemOverlayStyle: SystemUiOverlayStyle(
          statusBarColor: Colors.white70,
          statusBarIconBrightness: Brightness.light,
        ),
        backgroundColor: Colors.white,
        elevation: 0,
        titleTextStyle: TextStyle(
          color: Colors.black,
          fontWeight: FontWeight.bold,
          fontSize: 20.0,
        ),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        type: BottomNavigationBarType.fixed,
        selectedItemColor: Colors.deepOrange,
        unselectedItemColor: Colors.grey,
        elevation: 0,
        backgroundColor: Colors.white,
      ),
    ),
    darkTheme: ThemeData(
      primarySwatch: primarySwatch,
      scaffoldBackgroundColor: const Color(0xff0e1117),
      appBarTheme: const AppBarTheme(
        iconTheme: IconThemeData(
          color: Colors.white,
        ),
        actionsIconTheme: IconThemeData(
          color: Colors.white,
        ),
        systemOverlayStyle: SystemUiOverlayStyle(
          statusBarColor: Colors.transparent,
          statusBarIconBrightness: Brightness.dark,
        ),
        backgroundColor: Color(0xff0e1117),
        elevation: 0,
        titleTextStyle: TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.bold,
          fontSize: 20.0,
        ),
      ),
      bottomNavigationBarTheme: const BottomNavigationBarThemeData(
        type: BottomNavigationBarType.fixed,
        selectedItemColor: Colors.deepOrange,
        unselectedItemColor: Colors.grey,
        elevation: 0,
        backgroundColor: Color(0xff0e1117),
      ),
    ),
    themeMode: ThemeMode.light,
    home: const HomePage(),
    routes: {
      loginRoute: (context) => const LoginView(),
      registerRoute: (context) => const RegisterView(),
      homepageRoute: (context) => const HomePage(),
      homeRoute: (context) => const HomeLayout(),
      verifyRoute: (context) => const VerifyEmailView(),
    },
  ));
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return FutureBuilder(
      //initializing the connection with firebase
      future: AuthService.firebase().initialize(),
      builder: (context, snapshot) {
        //switch to check wether the connection stablished or not.
        switch (snapshot.connectionState) {
          case ConnectionState.done:
            final user = AuthService.firebase().currentUser;
            if (user != null) {
              if (user.isEmailVerified) {
                return const SplashScreen(view: HomeLayout());
              } else {
                return const SplashScreen(view: VerifyEmailView());
              }
            } else {
              return const SplashScreen(view: LoginView());
            }
          default:
            return const Scaffold(
              body: CircularProgressIndicator(),
            );
        }
      },
    );
  }
}
