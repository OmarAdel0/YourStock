import 'dart:async';

import 'package:flutter/material.dart';
// import 'package:yourstock/views/login_view.dart';

class SplashScreen extends StatefulWidget {
  final Widget view;
  const SplashScreen({Key? key, required this.view}) : super(key: key);

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Timer(
        const Duration(seconds: 3),
        () => Navigator.pushReplacement(
            context, MaterialPageRoute(builder: (context) => widget.view)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          color: Colors.white,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            Image.asset(
              'assets/images/logos/start_screen.png',
              width: 400.0,
              height: 400.0,
            ),
          ],
        ),
      ),
    );
  }
}
