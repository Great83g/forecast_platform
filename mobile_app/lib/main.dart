import 'package:flutter/material.dart';

void main() {
  runApp(const ForecastMobileApp());
}

class ForecastMobileApp extends StatelessWidget {
  const ForecastMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Forecast Platform',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
      ),
      home: const _HomeScreen(),
    );
  }
}

class _HomeScreen extends StatelessWidget {
  const _HomeScreen();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Forecast Mobile'),
      ),
      body: const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            'Базовый Flutter-клиент создан. Следующий шаг — подключение к API портала и экран авторизации.',
            textAlign: TextAlign.center,
          ),
        ),
      ),
    );
  }
}
