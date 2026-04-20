import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import 'core/api_config.dart';

void main() {
  runApp(const ForecastMobileApp());
}

class ForecastMobileApp extends StatelessWidget {
  const ForecastMobileApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Forecast Platform',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF4F46E5)),
        useMaterial3: true,
      ),
      home: const PortalLauncherScreen(),
    );
  }
}

class PortalLauncherScreen extends StatefulWidget {
  const PortalLauncherScreen({super.key});

  @override
  State<PortalLauncherScreen> createState() => _PortalLauncherScreenState();
}

class _PortalLauncherScreenState extends State<PortalLauncherScreen> {
  bool _isOpening = false;

  Uri get _portalUri => Uri.parse(ApiConfig.baseUrl);

  Future<void> _openPortal() async {
    setState(() => _isOpening = true);

    try {
      final opened = await launchUrl(
        _portalUri,
        mode: LaunchMode.externalApplication,
      );

      if (!opened && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Не удалось открыть: ${_portalUri.toString()}')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка открытия: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _isOpening = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Forecast Platform')),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Icon(Icons.language, size: 72),
              const SizedBox(height: 16),
              Text(
                'Кардинальный режим: открытие портала в браузере',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 12),
              Text(
                'URL: ${_portalUri.toString()}',
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              FilledButton.icon(
                onPressed: _isOpening ? null : _openPortal,
                icon: _isOpening
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.open_in_new),
                label: Text(_isOpening ? 'Открываю...' : 'Открыть портал'),
              ),
              const SizedBox(height: 12),
              const Text(
                'Это самый стабильный путь прямо сейчас: авторизация работает как на сайте, без проблем WebView/эмулятора.',
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
