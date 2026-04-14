class ApiConfig {
  const ApiConfig._();

  static const String _defaultBaseUrl = 'http://10.0.2.2:8000';

  /// Base URL for backend API.
  ///
  /// Override at runtime:
  /// flutter run --dart-define=API_BASE_URL=http://<server-ip>:<port>
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: _defaultBaseUrl,
  );
}
