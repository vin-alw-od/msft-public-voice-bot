using EchoBot;
using Microsoft.Extensions.Logging.Configuration;
using Microsoft.Extensions.Logging.EventLog;
using Azure.Extensions.AspNetCore.Configuration.Secrets;
using Azure.Identity;

IHost host;
try
{
    host = Host.CreateDefaultBuilder(args)
        .UseWindowsService(options =>
        {
            options.ServiceName = "Echo Bot Service";
        })
        .ConfigureAppConfiguration((context, config) =>
        {
            // Add Azure Key Vault configuration
            var keyVaultEndpoint = "https://acu1-tmagent-bot-d1-kv.vault.azure.net/";
            
            try
            {
                config.AddAzureKeyVault(
                    new Uri(keyVaultEndpoint),
                    new DefaultAzureCredential(),
                    new AzureKeyVaultConfigurationOptions
                    {
                        ReloadInterval = TimeSpan.FromHours(1) // Optional: reload secrets periodically
                    }
                );
            }
            catch (Exception ex)
            {
                // Log the error but don't crash - fallback to appsettings
                var logger = LoggerFactory.Create(builder => builder.AddConsole()).CreateLogger("Program");
                logger.LogWarning($"Failed to connect to Key Vault: {ex.Message}. Falling back to appsettings.json and environment variables");
            }
        })
        .ConfigureServices(services =>
        {
            LoggerProviderOptions.RegisterProviderOptions<
                EventLogSettings, EventLogLoggerProvider>(services);

            services.AddSingleton<IBotHost, BotHost>();

            services.AddHostedService<EchoBotWorker>();
        })
        .Build();
}
catch (Exception ex)
{
    // Fallback: create host without Key Vault if it fails
    var logger = LoggerFactory.Create(builder => builder.AddConsole()).CreateLogger("Program");
    logger.LogError($"Failed to build host with Key Vault: {ex.Message}. Building without Key Vault integration.");
    
    host = Host.CreateDefaultBuilder(args)
        .UseWindowsService(options =>
        {
            options.ServiceName = "Echo Bot Service";
        })
        .ConfigureServices(services =>
        {
            LoggerProviderOptions.RegisterProviderOptions<
                EventLogSettings, EventLogLoggerProvider>(services);

            services.AddSingleton<IBotHost, BotHost>();

            services.AddHostedService<EchoBotWorker>();
        })
        .Build();
}

await host.RunAsync();
