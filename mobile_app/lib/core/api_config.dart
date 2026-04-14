class ApiConfig {
  const ApiConfig._();

  static const String _defaultBaseUrl = 'http://10.0.2.2:8000';
  static const String _defaultLoginPath = '/auth/login';

  /// Base URL for backend API.
  ///
  /// Override at runtime:
  /// flutter run --dart-define=API_BASE_URL=http://<server-ip>:<port>
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: _defaultBaseUrl,
  );

  /// Backend login endpoint path.
  ///
  /// Override at runtime:
  /// flutter run --dart-define=LOGIN_PATH=/api/login
  static const String loginPath = String.fromEnvironment(
    'LOGIN_PATH',
    defaultValue: _defaultLoginPath,
  );
}
